package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

class Order_equals_0_0_Test {

    @Test
    void testEquals() {
        Order order1 = new Order();
        Order order2 = new Order();
        // All fields are null
        assertTrue(order1.equals(order2));
        // All fields are not null
        order1.m_orderId = 1;
        order1.m_clientId = 2;
        order1.m_totalQuantity = 3;
        order1.m_lmtPrice = 4.0;
        order1.m_auxPrice = 5.0;
        order1.m_ocaType = 6;
        order1.m_transmit = true;
        order1.m_parentId = 7;
        order1.m_blockOrder = true;
        order1.m_sweepToFill = true;
        order1.m_displaySize = 8;
        order1.m_triggerMethod = 9;
        order1.m_outsideRth = true;
        order1.m_hidden = true;
        order1.m_overridePercentageConstraints = true;
        order1.m_allOrNone = true;
        order1.m_minQty = 10;
        order1.m_percentOffset = 11.0;
        order1.m_trailStopPrice = 12.0;
        order1.m_origin = 13;
        order1.m_shortSaleSlot = 14;
        order1.m_discretionaryAmt = 15.0;
        order1.m_eTradeOnly = true;
        order1.m_firmQuoteOnly = true;
        order1.m_nbboPriceCap = 16.0;
        order1.m_auctionStrategy = 17;
        order1.m_startingPrice = 18.0;
        order1.m_stockRefPrice = 19.0;
        order1.m_delta = 20.0;
        order1.m_stockRangeLower = 21.0;
        order1.m_stockRangeUpper = 22.0;
        order1.m_volatility = 23.0;
        order1.m_volatilityType = 24;
        order1.m_continuousUpdate = 25;
        order1.m_referencePriceType = 26;
        order1.m_deltaNeutralAuxPrice = 27.0;
        order1.m_basisPoints = 28.0;
        order1.m_basisPointsType = 29;
        order1.m_scaleInitLevelSize = 30;
        order1.m_scaleSubsLevelSize = 31;
        order1.m_scalePriceIncrement = 32.0;
        order1.m_account = "account";
        order1.m_settlingFirm = "settlingFirm";
        order1.m_clearingAccount = "clearingAccount";
        order1.m_clearingIntent = "clearingIntent";
        order1.m_algoStrategy = "algoStrategy";
        order1.m_algoParams = new Vector();
        order1.m_whatIf = true;
        order1.m_notHeld = true;
        order2.m_orderId = 1;
        order2.m_clientId = 2;
        order2.m_totalQuantity = 3;
        order2.m_lmtPrice = 4.0;
        order2.m_auxPrice = 5.0;
        order2.m_ocaType = 6;
        order2.m_transmit = true;
        order2.m_parentId = 7;
        order2.m_blockOrder = true;
        order2.m_sweepToFill = true;
        order2.m_displaySize = 8;
        order2.m_triggerMethod = 9;
        order2.m_outsideRth = true;
        order2.m_hidden = true;
        order2.m_overridePercentageConstraints = true;
        order2.m_allOrNone = true;
    }
}
