package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_marketCapAbove_8_0_Test {

    @Test
    void testMarketCapAbove_positiveValue() {
        ScannerSubscription subscription = new ScannerSubscription();
        double expectedMarketCap = 1000.50;
        try {
            Field marketCapField = ScannerSubscription.class.getDeclaredField("m_marketCapAbove");
            marketCapField.setAccessible(true);
            marketCapField.set(subscription, expectedMarketCap);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Error accessing private field: " + e.getMessage());
        }
        double actualMarketCap = subscription.marketCapAbove();
        assertEquals(expectedMarketCap, actualMarketCap);
    }

    @Test
    void testMarketCapAbove_zeroValue() {
        ScannerSubscription subscription = new ScannerSubscription();
        double expectedMarketCap = 0;
        try {
            Field marketCapField = ScannerSubscription.class.getDeclaredField("m_marketCapAbove");
            marketCapField.setAccessible(true);
            marketCapField.set(subscription, expectedMarketCap);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Error accessing private field: " + e.getMessage());
        }
        double actualMarketCap = subscription.marketCapAbove();
        assertEquals(expectedMarketCap, actualMarketCap);
    }

    @Test
    void testMarketCapAbove_maxValue() {
        ScannerSubscription subscription = new ScannerSubscription();
        try {
            Field marketCapField = ScannerSubscription.class.getDeclaredField("m_marketCapAbove");
            marketCapField.setAccessible(true);
            marketCapField.set(subscription, Double.MAX_VALUE);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Error accessing private field: " + e.getMessage());
        }
        double actualMarketCap = subscription.marketCapAbove();
        assertEquals(Double.MAX_VALUE, actualMarketCap);
    }

    @Test
    void testMarketCapAbove_minValue() {
        ScannerSubscription subscription = new ScannerSubscription();
        try {
            Field marketCapField = ScannerSubscription.class.getDeclaredField("m_marketCapAbove");
            marketCapField.setAccessible(true);
            marketCapField.set(subscription, Double.MIN_VALUE);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Error accessing private field: " + e.getMessage());
        }
        double actualMarketCap = subscription.marketCapAbove();
        assertEquals(Double.MIN_VALUE, actualMarketCap);
    }

    @Test
    void testMarketCapAbove_default() {
        ScannerSubscription subscription = new ScannerSubscription();
        double actualMarketCap = subscription.marketCapAbove();
        assertEquals(Double.MAX_VALUE, actualMarketCap);
    }
}
