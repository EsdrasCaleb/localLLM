package com.ib.client;

import java.util.ArrayList;
import java.util.List;
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
    public void testOpenOrder() {
        // Arrange
        Contract contract = new Contract();
        Order order = new Order();
        OrderState orderState = new OrderState();
        EWrapperMsgGenerator msgGenerator = new EWrapperMsgGenerator();
        // Act
        String msg = msgGenerator.openOrder(1, contract, order, orderState);
        // Assert
        assertEquals("open order: orderId=1 action=1 quantity=1000 symbol=ABC exchange=USD symbol=ABC exchange=USD secType=STK type=STK lmtPrice=1000.00 auxPrice=100.00 TIF=1.00 localSymbol=ABC localSymbol=ABC client Id=1234 parent Id=1234 permId=1234 outsideRth=1.00 hidden=1 discretionaryAmt=100.00 triggerMethod=1 goodAfterTime=1.00 goodTillDate=1.00 faGroup=1 faMethod=1 faPercentage=1.00 faProfile=1 shortSaleSlot=1 ocaGroup=1 ocaType=1 rule80A=1 allOrNone=1 minQty=100.00 percentOffset=1.00 eTradeOnly=1 firmQuoteOnly=1 nbboPriceCap=10000.00 auctionStrategy=1 startingPrice=100.00 stockRefPrice=100.00 delta=1.00 stockRangeLower=0.00 stockRangeUpper=100.00 volatility=1.00 volatilityType=1 deltaNeutralOrderType=1 deltaNeutralAuxPrice=1.00 continuousUpdate=1 referencePriceType=1 trailStopPrice=1.00 scaleInitLevelSize=100.00 scaleSubsLevelSize=100.00 scalePriceIncrement=1.00 account=1234 settlingFirm=1234 clearingAccount=1234 clearingIntent=1 notHeld=0.00 whatIf=1", msg);
        assertTrue(msg.startsWith("open order:"));
        assertTrue(msg.contains("orderId=1"));
        assertTrue(msg.contains("action=1"));
        assertTrue(msg.contains("quantity=1000"));
        assertTrue(msg.contains("symbol=ABC"));
        assertTrue(msg.contains("exchange=USD"));
        assertTrue(msg.contains("secType=STK"));
        assertTrue(msg.contains("type=STK"));
        assertTrue(msg.contains("lmtPrice=1000.00"));
        assertTrue(msg.contains("auxPrice=100.00"));
        assertTrue(msg.contains("TIF=1.00"));
        assertTrue(msg.contains("localSymbol=ABC"));
        assertTrue(msg.contains("client Id=1234"));
        assertTrue(msg.contains("parent Id=1234"));
        assertTrue(msg.contains("permId=1234"));
        assertTrue(msg.contains("outsideRth=1.00"));
        assertTrue(msg.contains("hidden=1"));
        assertTrue(msg.contains("discretionaryAmt=100.00"));
        assertTrue(msg.contains("triggerMethod=1"));
        assertTrue(msg.contains("goodAfterTime=1.00"));
        assertTrue(msg.contains("goodTillDate=1.00"));
        assertTrue(msg.contains("faGroup=1"));
        assertTrue(msg.contains("faMethod=1"));
        assertTrue(msg.contains("faPercentage=1.00"));
        assertTrue(msg.contains("faProfile=1"));
        assertTrue(msg.contains("shortSaleSlot=1"));
        assertTrue(msg.contains("ocaGroup=1"));
        assertTrue(msg.contains("ocaType=1"));
        assertTrue(msg.contains("rule80A=1"));
        assertTrue(msg.contains("allOrNone=1"));
        assertTrue(msg.contains("minQty=100.00"));
        assertTrue(msg.contains("percentOffset=1.00"));
        assertTrue(msg.contains("eTradeOnly=1"));
        assertTrue(msg.contains("firmQuoteOnly=1"));
        assertTrue(msg.contains("nbboPriceCap=10000.00"));
        assertTrue(msg.contains("auctionStrategy=1"));
        assertTrue(msg.contains("startingPrice=100.00"));
        assertTrue(msg.contains("stockRefPrice=100.00"));
        assertTrue(msg.contains("delta=1.00"));
        assertTrue(msg.contains("stockRangeLower=0.00"));
        assertTrue(msg.contains("stockRangeUpper=100.00"));
        assertTrue(msg.contains("volatility=1.00"));
        assertTrue(msg.contains("volatilityType=1"));
        assertTrue(msg.contains("deltaNeutralOrderType=1"));
        assertTrue(msg.contains("deltaNeutralAuxPrice=1.00"));
        assertTrue(msg.contains("continuousUpdate=1"));
        assertTrue(msg.contains("referencePriceType=1"));
        assertTrue(msg.contains("trailStopPrice=1.00"));
    }
}
