using Microsoft.Playwright;
using DotNetEnv;

namespace PlaywrightXBot;

class Program
{
    static async Task Main(string[] args)
    {
        // Install browsers programmatically if needed
        Console.WriteLine("Ensuring browsers are installed...");
        Microsoft.Playwright.Program.Main(new[] { "install" });

        // Load environment variables from .env file
        Env.Load();

        var username = Environment.GetEnvironmentVariable("X_USERNAME");
        var password = Environment.GetEnvironmentVariable("X_PASSWORD");

        if (string.IsNullOrEmpty(username) || string.IsNullOrEmpty(password))
        {
            Console.WriteLine("Error: X_USERNAME and X_PASSWORD must be set in .env file");
            return;
        }

        // Initialize Playwright
        using var playwright = await Playwright.CreateAsync();

        // Launch browser (headless mode for WSL without desktop)
        /* Trying to see in regular mode now
        await using var browser = await playwright.Chromium.LaunchAsync(new BrowserTypeLaunchOptions
        {
            Headless = true, // Set to false if you want to see the browser (requires X11 forwarding)
            Args = new[] { "--no-sandbox", "--disable-dev-shm-usage" } // WSL-friendly options
        });
        */

        await using var browser = await playwright.Chromium.LaunchAsync(new BrowserTypeLaunchOptions
        {
            Headless = false,
            SlowMo = 500,
            Args = new[] {
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-web-security", // may help with WSL
                "--disable-feature=VizDisplayCompositor" // help with display issues
            }
        });

        // Creating browser context w/video recording
        /* I'm low on space
        var context = await browser.NewContextAsync(new BrowserNewContextOptions
        {
            RecordVideoDir = "./videos/",
            RecordVideoSize = new RecordVideoSize() { Width = 1280, Height = 720 }
        });
        */

        // Create a new page
        var page = await browser.NewPageAsync();

        try
        {
            Console.WriteLine("Navigating to X.com...");
            await page.GotoAsync("https://x.com/login");

            // Wait for the login form to load
            await page.WaitForSelectorAsync("input[name='text']", new PageWaitForSelectorOptions
            {
                Timeout = 10000
            });

            Console.WriteLine("Entering username...");
            // Enter username
            await page.FillAsync("input[name='text']", username);
            await page.ClickAsync("xpath=//span[text()='Next']/..");

            // Wait for password field
            await page.WaitForSelectorAsync("input[name='password']", new PageWaitForSelectorOptions
            {
                Timeout = 10000
            });

            Console.WriteLine("Entering password...");
            // Enter password
            await page.FillAsync("input[name='password']", password);
            await page.ClickAsync("xpath=//span[text()='Log in']/..");

            // Wait for successful login (look for home timeline or user menu)
            try
            {
                await page.WaitForSelectorAsync("xpath=//span[text()='Home']", new PageWaitForSelectorOptions
                {
                    Timeout = 15000
                });
                Console.WriteLine("Successfully logged in to X.com!");

                // Get current URL to confirm we're logged in
                var currentUrl = page.Url;
                Console.WriteLine($"Current URL: {currentUrl}");

                // Optional: Take a screenshot to verify login
                /* Cool but I don't want this ATM
                await page.ScreenshotAsync(new PageScreenshotOptions
                {
                    Path = "login_success.png",
                    FullPage = true
                });
                Console.WriteLine("Screenshot saved as login_success.png");
                */

                // Wait a moment before closing
                // NOT CLOSING Right now
                // await page.WaitForTimeoutAsync(3000);
            }
            catch (TimeoutException)
            {
                Console.WriteLine("Login may have failed - couldn't find home timeline");

                // Check if we need to handle additional verification
                var currentUrl = page.Url;
                Console.WriteLine($"Current URL after login attempt: {currentUrl}");

                // Take screenshot for debugging
                await page.ScreenshotAsync(new PageScreenshotOptions
                {
                    Path = "login_failed.png",
                    FullPage = true
                });
                Console.WriteLine("Debug screenshot saved as login_failed.png");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred: {ex.Message}");

            // Take screenshot for debugging
            await page.ScreenshotAsync(new PageScreenshotOptions
            {
                Path = "error_screenshot.png",
                FullPage = true
            });
            Console.WriteLine("Error screenshot saved as error_screenshot.png");
        }
        finally
        {
            Console.WriteLine("Closing browser...");
        }
    }
}
