package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Order_equals_0_0_Test {

    @Test
    public void testEquals() {
        Order order1 = new Order();
        Order order2 = new Order();
        order1.m_orderId = 1;
        order1.m_clientId = 1;
        order1.m_permId = 1;
        order1.m_action = "action";
        order1.m_orderType = "orderType";
        order1.m_tif = "tif";
        order1.m_ocaGroup = "ocaGroup";
        order1.m_orderRef = "orderRef";
        order1.m_goodAfterTime = "goodAfterTime";
        order1.m_goodTillDate = "goodTillDate";
        order1.m_rule80A = "rule80A";
        order1.m_faGroup = "faGroup";
        order1.m_faProfile = "faProfile";
        order1.m_faMethod = "faMethod";
        order1.m_faPercentage = "faPercentage";
        order1.m_openClose = "openClose";
        order1.m_designatedLocation = "designatedLocation";
        order1.m_deltaNeutralOrderType = "deltaNeutralOrderType";
        order1.m_account = "account";
        order1.m_settlingFirm = "settlingFirm";
        order1.m_clearingAccount = "clearingAccount";
        order1.m_clearingIntent = "clearingIntent";
        order1.m_algoStrategy = "algoStrategy";
        order1.m_algoParams = new Vector<>();
        order2.m_orderId = 1;
        order2.m_clientId = 1;
        order2.m_permId = 1;
        order2.m_action = "action";
        order2.m_orderType = "orderType";
        order2.m_tif = "tif";
        order2.m_ocaGroup = "ocaGroup";
        order2.m_orderRef = "orderRef";
        order2.m_goodAfterTime = "goodAfterTime";
        order2.m_goodTillDate = "goodTillDate";
        order2.m_rule80A = "rule80A";
        order2.m_faGroup = "faGroup";
        order2.m_faProfile = "faProfile";
        order2.m_faMethod = "faMethod";
        order2.m_faPercentage = "faPercentage";
        order2.m_openClose = "openClose";
        order2.m_designatedLocation = "designatedLocation";
        order2.m_deltaNeutralOrderType = "deltaNeutralOrderType";
        order2.m_account = "account";
        order2.m_settlingFirm = "settlingFirm";
        order2.m_clearingAccount = "clearingAccount";
        order2.m_clearingIntent = "clearingIntent";
        order2.m_algoStrategy = "algoStrategy";
        order2.m_algoParams = new Vector<>();
        assertEquals(true, order1.equals(order2));
    }
}
