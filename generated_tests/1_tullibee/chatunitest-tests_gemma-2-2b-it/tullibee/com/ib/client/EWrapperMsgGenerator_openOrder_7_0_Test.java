package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

public class EWrapperMsgGenerator_openOrder_7_0_Test {

    @Test
    void openOrder() {
        int orderId = 1;
        Contract contract = mock(Contract.class);
        Order order = mock(Order.class);
        OrderState orderState = mock(OrderState.class);
        EWrapperMsgGenerator msgGenerator = new EWrapperMsgGenerator();
        String actualMsg = msgGenerator.openOrder(orderId, contract, order, orderState);
        String expectedMsg = "open order: orderId=1 action=BUY quantity=1 symbol=AAPL exchange=NASDAQ secType=STK type=MARKET lmtPrice=1000 auxPrice=1000 TIF=0 localSymbol=AAPL client Id=1 parent Id=1 permId=1 outsideRth=0 hidden=0 discretionaryAmt=0 triggerMethod=0 goodAfterTime=0 goodTillDate=0 faGroup=0 faMethod=0 faPercentage=0 faProfile=0 shortSaleSlot=0 designatedLocation=0 ocaGroup=0 ocaType=0 rule80A=0 allOrNone=0 minQty=0 percentOffset=0 eTradeOnly=0 firmQuoteOnly=0 nbboPriceCap=0 auctionStrategy=0 startingPrice=0 stockRefPrice=0 delta=0 stockRangeLower=0 stockRangeUpper=0 volatility=0 volatilityType=0 deltaNeutralOrderType=0 deltaNeutralAuxPrice=0 continuousUpdate=0 referencePriceType=0 trailStopPrice=0 scaleInitLevelSize=100 scaleSubsLevelSize=100 scalePriceIncrement=0 account=1 settlingFirm=1 clearingAccount=1 clearingIntent=1 notHeld=0 whatIf=0";
        assertEquals(expectedMsg, actualMsg);
    }
}
