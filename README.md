# Finance Application
https://sparikh099-finance-testing-iiljow.streamlitapp.com/ 

This application is designed to provide you with stock analysis of several stocks of your choosing
through options pricing and indicators of stocks. 
Some of the the things that are included within this application include the 
 - Black-Scholes Model for Option Pricing for both Puts and Calls
 - A Taylor approximation of the Black-Scholes Model for Puts and Calls
 - Company Information
 - Latest News Articles

# What are Options?
Options are contracts that give the purchaser the right, but not the obligation to buy or sell a specific 
security whether it is a stock or an ETF at a fixed price with a specific period of time. 
In Simple Words, options allow you to bet if a stock is going up or down over a period of time. 
There are four basic types of options
- Buying a Call Option
- Selling a Call Option
- Calling a Put Option
- Selling a Put Option
This application is designed to price options. 

## European Options
A European Option gives the option holder the right to exercise the Option only at a pre-agreed future data and price.  

## American Options
An American Option gives the option holder the right to exercise the Option at any date before the expiration date at the pre-agreed price. 

# What is the Black-Scholes Model?
The Black-Scholes Model, also known as the Black-Scholes Merton Formula, is a differential equation that is used to price option contracts. 
The model is able to help declare the fair price or theoretical value for a call or put option. This model also uses the concept of random walk
and Geometric Brownian motion.
The random walk theory suggests that changes in stock prices have the same distribution and are independent of each other. 
Based on this theory, we cannot predict the future movement of a stock. 
The formula for the 

There are 5 important variables that need to be inputted in order to use the pricing model.
- The Stike Price of the Option
- The Current Stock Price
- The Time to Expiration in Years
- The Risk-Free-Rate
- The volatility

In this Black-Scholes Calculator, you will only need to input the 
Ticker of the Stock
Strike Price
Time to Expiration in Years
The Risk-Free-Rate

The calculator will be able to use the ticker of the stock to find the stock price and the volatility. 

## Assumptions of The Black-Scholes Model
- The Option is European 
- The Option could only be exercised at expiration. 
- No dividends are paid during the life of the option
    - Dividends are a sum of money paid by a company to its shareholders of its profits. Companies have the option to give dividends out to 
      its shareholders
- Market Movements cannot be predicted
- There are no transaction costs in buying the option
- The Risk Free Rate and Volatility are constant
- The Returns on the underlying asset are log normally distributed


## Limitations of The Black-Scholes Model
- This is only meant for European Options. We cannot use the Black-Scholes Model for American Options.
    - We cannot use American Options because American Options can be exercised before the expiration date.
- The model assumes that volatility, risk-free-rate, and dividends are constant over the option's life.
- Not taking to account taxes, commissions, or trading costs or taxes can also lead to valuations that are different from real-world results.

### Who were the founders of the Black-Scholes Model?
- Robert C. Merton - an American Economist and a professor at the MIT Sloan School of Management. He won the Nobel Memorial Prize for Economic Sciences after his contributions to the Black-Scholes Model.  
- Myron Scholes - a Canadian-American Economist and a professor of Finance at the Stanford Graduate School of Business. Similar to Merton, he won the Nobel Memorial Prize for Economic Sciences due to his contributions to the options pricing model. 
- Fischer Black - was also a major contributor to the Black-Scholes Model. He would have likely won a Nobel Prize along Scholes and Merton if not for his death in 1995. Fischer Black has had a 25-year Partnership with Robert C. Merton at the Massachusetts Institute of Technology.

These three people were able to really help revolutionize the world of quantitative finance. 


## Ito's Lemma
Brownian Motion is the main concept that the Black-Scholes Model is derived from. A Geometric Brownian Motion is an example of a simple random walk that is commonly used in the applications of finance. This motion suggests that we cannot predict the future movement of a financial asset. 
 
A Brownian motion with drift and diffusion satisfies the following Stochastic Differential Equation where μ and σ are constants. 

dX(t) = μdt + σdB(t)

The Drift and Diffusion coefficients can be functions of X(t) and t rather than any constants. 

Drift Coefficients  - a(X(t),t) = μ
Diffusion Coefficients - b(X(t),t) = σ

Down Below is the formula for the Ito drift-diffusion process:

dX(t) = a(X(t),t)dt + b(X(t),t)dB(t)

Using Ito's Lemma, we derive the function:

df = ((∂f/∂t) + (∂f/∂X)a + 1/2(∂^2f/∂X^2)b^2)dt + (∂f/∂X)bdB  

Based on looking at both of the functions, we can tell that both functions are affected by an underlying source of uncertainty,dB.
The randomness that is used in the Stochastic processes of X and f come from the same type of Brownian motion. 
This level of uncertainty plays a factor in the derivation of the Black-Scholes Formula. 

When we Apply Ito's Lemma into the partial derivative of C(S,t) where C is equal to the price of the option and S is the underlying stock price and t is equal to the time. We get the following equation:

dC = ((∂C/∂S)μS + (∂C/∂t) + 1/2(∂^2C/∂S^2)σ^2S^2)dt + (∂C/∂S)σSdB

Using this equation, it can be told that the change in the price of the option and the change in the underlying stock price is derived from the same geometric Brownian motion. Due to this, we can cancel out the factor of Brownian motion and simple random walk. 
Based on this analysis, the Black-Scholes Differential Equation was calculated.

rC = (∂C/∂t) + rS(∂C/∂S)  +1/2(∂^2C/∂S^2)σ^2S^2

The value of r represents the risk-free rate of the market. In an arbitrage free market, the portfolio must earn this value of r. 
The term arbitrage refers to the sales of multiple assets in different markets in order to make a profit from a tiny difference of an asset's listed price. This equation is not the Black-Scholes Option Pricing Model itself, but it displays the reasoning behind the model.


## Bibliography
Shi, Chuan. “Brownian Motion, Ito's Lemma, and the Black-Scholes Formula (Part II).” LinkedIn, LinkedIn, 8 June 2019, https://www.linkedin.com/pulse/brownian-motion-itos-lemma-black-scholes-formula-part-chuan-shi-1d. 
Smith, Tim. “Random Walk Theory.” Investopedia, Investopedia, 8 Feb. 2022, https://www.investopedia.com/terms/r/randomwalktheory.asp. 
Hayes, Adam. “What Is the Black-Scholes Model?” Investopedia, Investopedia, 12 June 2022, https://www.investopedia.com/terms/b/blackscholes.asp#:~:text=The%20Black%2DScholes%20model%2C%20aka,free%20rate%2C%20and%20the%20volatility. 
Yoo, Younggeun. University of Chicago, 2017, Stochastic Calculus and Black-Scholes Model, https://math.uchicago.edu/~may/REU2017/REUPapers/Yoo.pdf. Accessed 12 June 2022. 
“Black-Scholes Formula (D1, D2, Call Price, Put Price, Greeks).” Macroption, 2022, https://www.macroption.com/black-scholes-formula/. 





