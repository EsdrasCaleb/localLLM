// Test method
package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

public class EWrapperMsgGenerator_openOrder_7_1_Test {

    @ExtendWith(MockitoExtension.class)
    public class TestClass {

        @Test
        public void testOpenOrder() {
            // Arrange
            Contract contract = Mockito.mock(Contract.class);
            Order order = Mockito.mock(Order.class);
            OrderState orderState = Mockito.mock(OrderState.class);
            Mockito.when(contract.m_secType).thenReturn("BAG");
            Mockito.when(contract.m_comboLegsDescrip).thenReturn(null);
            Mockito.when(order.m_basisPoints).thenReturn(Double.MAX_VALUE);
            Mockito.when(order.m_algoStrategy).thenReturn("algoStrategy");
            Mockito.when(order.m_algoParams).thenReturn(new Vector());
            Mockito.when(order.m_algoParams.get(0)).thenReturn(new TagValue("tag1", "value1"));
            Mockito.when(order.m_algoParams.get(1)).thenReturn(new TagValue("tag2", "value2"));
            Mockito.when(order.m_algoParams.get(2)).thenReturn(new TagValue("tag3", "value3"));
            // Act
            String result = EWrapperMsgGenerator.openOrder(1, contract, order, orderState);
            // Assert
            assertEquals("open order: orderId=1 action=BAG quantity=0 symbol=BAG exchange=BAG secType=BAG type=BUY lmtPrice=0 auxPrice=0 TIF=NONE localSymbol=BAG client Id=1 parent Id=1 permId=1 outsideRth=false hidden=false discretionaryAmt=0 triggerMethod=BUY goodAfterTime=0 goodTillDate=0 faGroup=0 faMethod=0 faPercentage=0 faProfile=0 shortSaleSlot=0 designatedLocation=0 ocaGroup=0 ocaType=0 rule80A=false allOrNone=false minQty=0 percentOffset=0 eTradeOnly=false firmQuoteOnly=false nbboPriceCap=0 auctionStrategy=BUY startingPrice=0 stockRefPrice=0 delta=0 stockRangeLower=0 stockRangeUpper=0 volatility=0 volatilityType=NONE deltaNeutralOrderType=BUY deltaNeutralAuxPrice=0 continuousUpdate=false referencePriceType=NONE trailStopPrice=0 scaleInitLevelSize=0 scaleSubsLevelSize=0 scalePriceIncrement=0 account=0 settlingFirm=0 clearingAccount=0 clearingIntent=0 notHeld=0 whatIf=" + result, null);
        }
    }
}
