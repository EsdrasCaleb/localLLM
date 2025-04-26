package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ComboLeg_equals_0_0_Test {

    @Test
    void testEquals_sameObject() {
        ComboLeg leg = new ComboLeg(1, 2, "Buy", "NYSE", ComboLeg.OPEN);
        Assertions.assertTrue(leg.equals(leg));
    }

    @Test
    void testEquals_nullObject() {
        ComboLeg leg = new ComboLeg(1, 2, "Buy", "NYSE", ComboLeg.OPEN);
        Assertions.assertFalse(leg.equals(null));
    }

    @Test
    void testEquals_differentConId() {
        ComboLeg leg1 = new ComboLeg(1, 2, "Buy", "NYSE", ComboLeg.OPEN);
        ComboLeg leg2 = new ComboLeg(3, 2, "Buy", "NYSE", ComboLeg.OPEN);
        Assertions.assertFalse(leg1.equals(leg2));
    }

    @Test
    void testEquals_differentRatio() {
        ComboLeg leg1 = new ComboLeg(1, 2, "Buy", "NYSE", ComboLeg.OPEN);
        ComboLeg leg2 = new ComboLeg(1, 3, "Buy", "NYSE", ComboLeg.OPEN);
        Assertions.assertFalse(leg1.equals(leg2));
    }

    @Test
    void testEquals_differentOpenClose() {
        ComboLeg leg1 = new ComboLeg(1, 2, "Buy", "NYSE", ComboLeg.OPEN);
        ComboLeg leg2 = new ComboLeg(1, 2, "Buy", "NYSE", ComboLeg.CLOSE);
        Assertions.assertFalse(leg1.equals(leg2));
    }

    @Test
    void testEquals_differentShortSaleSlot() {
        ComboLeg leg1 = new ComboLeg(1, 2, "Buy", "NYSE", ComboLeg.OPEN, 1, "location");
        ComboLeg leg2 = new ComboLeg(1, 2, "Buy", "NYSE", ComboLeg.OPEN, 2, "location");
        Assertions.assertFalse(leg1.equals(leg2));
    }

    @Test
    void testEquals_differentAction() {
        ComboLeg leg1 = new ComboLeg(1, 2, "Buy", "NYSE", ComboLeg.OPEN);
        ComboLeg leg2 = new ComboLeg(1, 2, "sell", "NYSE", ComboLeg.OPEN);
        Assertions.assertFalse(leg1.equals(leg2));
    }

    @Test
    void testEquals_differentExchange() {
        ComboLeg leg1 = new ComboLeg(1, 2, "Buy", "NYSE", ComboLeg.OPEN);
        ComboLeg leg2 = new ComboLeg(1, 2, "Buy", "NASDAQ", ComboLeg.OPEN);
        Assertions.assertFalse(leg1.equals(leg2));
    }

    @Test
    void testEquals_differentDesignatedLocation() {
        ComboLeg leg1 = new ComboLeg(1, 2, "Buy", "NYSE", ComboLeg.OPEN, 1, "location1");
        ComboLeg leg2 = new ComboLeg(1, 2, "Buy", "NYSE", ComboLeg.OPEN, 1, "location2");
        Assertions.assertFalse(leg1.equals(leg2));
    }

    @Test
    void testEquals_allFieldsMatch() {
        ComboLeg leg1 = new ComboLeg(1, 2, "Buy", "NYSE", ComboLeg.OPEN, 1, "location");
        ComboLeg leg2 = new ComboLeg(1, 2, "buy", "NYSE", ComboLeg.OPEN, 1, "location");
        Assertions.assertTrue(leg1.equals(leg2));
    }
}
