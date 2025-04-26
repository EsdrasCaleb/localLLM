package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

class Order_equals_0_1_Test {

    @Test
    void testEquals() {
        Order order1 = new Order();
        order1.m_orderId = 1;
        order1.m_clientId = 1;
        order1.m_totalQuantity = 1;
        order1.m_lmtPrice = 1.0;
        order1.m_auxPrice = 1.0;
        order1.m_ocaType = 1;
        order1.m_transmit = true;
        order1.m_parentId = 1;
        order1.m_blockOrder = true;
        order1.m_sweepToFill = true;
        order1.m_displaySize = 1;
        order1.m_triggerMethod = 1;
        order1.m_outsideRth = true;
        order1.m_hidden = true;
        order1.m_goodAfterTime = "goodAfterTime";
        order1.m_goodTillDate = "goodTillDate";
        order1.m_overridePercentageConstraints = true;
        order1.m_rule80A = "rule80A";
        order1.m_allOrNone = true;
        order1.m_minQty = 1;
        order1.m_percentOffset = 1.0;
        order1.m_trailStopPrice = 1.0;
        order1.m_faGroup = "faGroup";
        order1.m_faProfile = "faProfile";
        order1.m_faMethod = "faMethod";
        order1.m_faPercentage = "faPercentage";
        order1.m_openClose = "openClose";
        order1.m_origin = 1;
        order1.m_shortSaleSlot = 1;
        order1.m_designatedLocation = "designatedLocation";
        order1.m_discretionaryAmt = 1.0;
        order1.m_eTradeOnly = true;
        order1.m_firmQuoteOnly = true;
        order1.m_nbboPriceCap = 1.0;
        order1.m_auctionStrategy = 1;
        order1.m_startingPrice = 1.0;
        order1.m_stockRefPrice = 1.0;
        order1.m_delta = 1.0;
        order1.m_stockRangeLower = 1.0;
        order1.m_stockRangeUpper = 1.0;
        order1.m_volatility = 1.0;
        order1.m_volatilityType = 1;
        order1.m_continuousUpdate = 1;
        order1.m_referencePriceType = 1;
        order1.m_deltaNeutralAuxPrice = 1.0;
        order1.m_basisPoints = 1.0;
        order1.m_basisPointsType = 1;
        order1.m_scaleInitLevelSize = 1;
        order1.m_scaleSubsLevelSize = 1;
        order1.m_scalePriceIncrement = 1.0;
        order1.m_whatIf = true;
        order1.m_notHeld = true;
        Order order2 = new Order();
        order2.m_orderId = 1;
        order2.m_clientId = 1;
        order2.m_totalQuantity = 1;
        order2.m_lmtPrice = 1.0;
        order2.m_auxPrice = 1.0;
        order2.m_ocaType = 1;
        order2.m_transmit = true;
        order2.m_parentId = 1;
        order2.m_blockOrder = true;
        order2.m_sweepToFill = true;
        order2.m_displaySize = 1;
        order2.m_triggerMethod = 1;
        order2.m_outsideRth = true;
        order2.m_hidden = true;
        order2.m_overridePercentageConstraints = true;
    }
}
