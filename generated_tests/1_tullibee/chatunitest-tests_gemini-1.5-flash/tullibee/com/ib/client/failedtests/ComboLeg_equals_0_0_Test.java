package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ComboLeg_equals_0_0_Test {

    // Dummy Util class for testing purposes.  Replace with your actual Util class.
    static class Util {

        public static int StringCompareIgnCase(String s1, String s2) {
            if (s1 == null && s2 == null)
                return 0;
            if (s1 == null)
                return -1;
            if (s2 == null)
                return 1;
            return s1.equalsIgnoreCase(s2) ? 0 : 1;
        }
    }

    @Test
    void testEqualsReflexive() {
        ComboLeg leg1 = new ComboLeg(1, 2, "BUY", "NYSE", ComboLeg.OPEN);
        assertTrue(leg1.equals(leg1));
    }

    @Test
    void testEqualsSymmetric() {
        ComboLeg leg1 = new ComboLeg(1, 2, "BUY", "NYSE", ComboLeg.OPEN);
        ComboLeg leg2 = new ComboLeg(1, 2, "BUY", "NYSE", ComboLeg.OPEN);
        assertTrue(leg1.equals(leg2) && leg2.equals(leg1));
    }

    @Test
    void testEqualsTransitive() {
        ComboLeg leg1 = new ComboLeg(1, 2, "BUY", "NYSE", ComboLeg.OPEN);
        ComboLeg leg2 = new ComboLeg(1, 2, "BUY", "NYSE", ComboLeg.OPEN);
        ComboLeg leg3 = new ComboLeg(1, 2, "BUY", "NYSE", ComboLeg.OPEN);
        assertTrue(leg1.equals(leg2) && leg2.equals(leg3) && leg1.equals(leg3));
    }

    @Test
    void testEqualsNull() {
        ComboLeg leg1 = new ComboLeg(1, 2, "BUY", "NYSE", ComboLeg.OPEN);
        assertFalse(leg1.equals(null));
    }

    @Test
    void testEqualsDifferentClass() {
        ComboLeg leg1 = new ComboLeg(1, 2, "BUY", "NYSE", ComboLeg.OPEN);
        assertFalse(leg1.equals("test"));
    }

    @Test
    void testEqualsDifferentConId() {
        ComboLeg leg1 = new ComboLeg(1, 2, "BUY", "NYSE", ComboLeg.OPEN);
        ComboLeg leg2 = new ComboLeg(2, 2, "BUY", "NYSE", ComboLeg.OPEN);
        assertFalse(leg1.equals(leg2));
    }

    @Test
    void testEqualsDifferentRatio() {
        ComboLeg leg1 = new ComboLeg(1, 2, "BUY", "NYSE", ComboLeg.OPEN);
        ComboLeg leg2 = new ComboLeg(1, 3, "BUY", "NYSE", ComboLeg.OPEN);
        assertFalse(leg1.equals(leg2));
    }

    @Test
    void testEqualsDifferentAction() {
        ComboLeg leg1 = new ComboLeg(1, 2, "BUY", "NYSE", ComboLeg.OPEN);
        ComboLeg leg2 = new ComboLeg(1, 2, "SELL", "NYSE", ComboLeg.OPEN);
        assertFalse(leg1.equals(leg2));
    }

    @Test
    void testEqualsDifferentActionCaseInsensitive() {
        ComboLeg leg1 = new ComboLeg(1, 2, "BUY", "NYSE", ComboLeg.OPEN);
        ComboLeg leg2 = new ComboLeg(1, 2, "buy", "NYSE", ComboLeg.OPEN);
        assertTrue(leg1.equals(leg2));
    }

    @Test
    void testEqualsDifferentExchange() {
        ComboLeg leg1 = new ComboLeg(1, 2, "BUY", "NYSE", ComboLeg.OPEN);
        ComboLeg leg2 = new ComboLeg(1, 2, "BUY", "NASDAQ", ComboLeg.OPEN);
        assertFalse(leg1.equals(leg2));
    }

    @Test
    void testEqualsDifferentExchangeCaseInsensitive() {
        ComboLeg leg1 = new ComboLeg(1, 2, "BUY", "NYSE", ComboLeg.OPEN);
        ComboLeg leg2 = new ComboLeg(1, 2, "BUY", "nyse", ComboLeg.OPEN);
    }
}
