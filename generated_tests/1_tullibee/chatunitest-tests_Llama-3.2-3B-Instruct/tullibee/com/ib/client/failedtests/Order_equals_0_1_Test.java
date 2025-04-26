package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class Order_equals_0_1_Test {

    @Mock
    private Order order1;

    @Mock
    private Order order2;

    @Mock
    private Vector algoParams;

    @InjectMocks
    private Order order3;

    @Test
    public void testEqualsOrderSameInstance() {
        assertTrue(order1.equals(order1));
    }

    @Test
    public void testEqualsOrderSameProperties() {
        when(order1.m_permId).thenReturn(1);
        when(order1.m_orderId).thenReturn(1);
        when(order1.m_clientId).thenReturn(1);
        when(order1.m_totalQuantity).thenReturn(1);
        when(order1.m_lmtPrice).thenReturn(1.0);
        when(order1.m_auxPrice).thenReturn(1.0);
        when(order1.m_ocaType).thenReturn(1);
        when(order1.m_transmit).thenReturn(true);
        when(order1.m_parentId).thenReturn(1);
        when(order1.m_blockOrder).thenReturn(true);
        when(order1.m_sweepToFill).thenReturn(true);
        when(order1.m_displaySize).thenReturn(1);
        when(order1.m_triggerMethod).thenReturn(1);
        when(order1.m_outsideRth).thenReturn(true);
        when(order1.m_hidden).thenReturn(true);
        when(order1.m_overridePercentageConstraints).thenReturn(true);
        when(order1.m_allOrNone).thenReturn(true);
        when(order1.m_minQty).thenReturn(1);
        when(order1.m_percentOffset).thenReturn(1.0);
        when(order1.m_trailStopPrice).thenReturn(1.0);
        when(order1.m_origin).thenReturn(1);
        when(order1.m_shortSaleSlot).thenReturn(1);
        when(order1.m_discretionaryAmt).thenReturn(1.0);
        when(order1.m_eTradeOnly).thenReturn(true);
        when(order1.m_firmQuoteOnly).thenReturn(true);
        when(order1.m_nbboPriceCap).thenReturn(1.0);
        when(order1.m_auctionStrategy).thenReturn(1);
        when(order1.m_startingPrice).thenReturn(1.0);
        when(order1.m_stockRefPrice).thenReturn(1.0);
        when(order1.m_delta).thenReturn(1.0);
        when(order1.m_stockRangeLower).thenReturn(1.0);
        when(order1.m_stockRangeUpper).thenReturn(1.0);
        when(order1.m_volatility).thenReturn(1.0);
        when(order1.m_volatilityType).thenReturn(1);
        when(order1.m_continuousUpdate).thenReturn(1);
        when(order1.m_referencePriceType).thenReturn(1);
        when(order1.m_deltaNeutralAuxPrice).thenReturn(1.0);
        when(order1.m_basisPoints).thenReturn(1.0);
        when(order1.m_basisPointsType).thenReturn(1);
        when(order1.m_scaleInitLevelSize).thenReturn(1);
        when(order1.m_scaleSubsLevelSize).thenReturn(1);
        when(order1.m_scalePriceIncrement).thenReturn(1.0);
        when(order1.m_algoStrategy).thenReturn("Strategy");
        when(order1.m_algoParams).thenReturn(algoParams);
        assertTrue(order1.equals(order2));
    }

    @Test
    public void testEqualsOrderDifferentProperties() {
        when(order1.m_permId).thenReturn(1);
        when(order1.m_orderId).thenReturn(1);
        when(order1.m_clientId).thenReturn(1);
        when(order1.m_totalQuantity).thenReturn(1);
        when(order1.m_lmtPrice).thenReturn(1.0);
        when(order1.m_auxPrice).thenReturn(1.0);
        when(order1.m_ocaType).thenReturn(1);
        when(order1.m_transmit).thenReturn(true);
        when(order1.m_parentId).thenReturn(1);
        when(order1.m_blockOrder).thenReturn(true);
        when(order1.m_sweepToFill).thenReturn(true);
        when(order1.m_displaySize).thenReturn(1);
        when(order1.m_triggerMethod).thenReturn(1);
        when(order1.m_outsideRth).thenReturn(true);
        when(order1.m_hidden).thenReturn(true);
        when(order1.m_overridePercentageConstraints).thenReturn(true);
        when(order1.m_allOrNone).thenReturn(true);
        when(order1.m_minQty).thenReturn(1);
        when(order1.m_percentOffset).thenReturn(1.0);
        when(order1.m_trailStopPrice).thenReturn(1.0);
    }
}
