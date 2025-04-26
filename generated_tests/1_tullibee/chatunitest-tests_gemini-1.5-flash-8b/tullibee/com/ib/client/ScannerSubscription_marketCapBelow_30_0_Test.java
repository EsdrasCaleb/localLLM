package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_marketCapBelow_30_0_Test {

    @Test
    void testMarketCapBelow() throws NoSuchFieldException, IllegalAccessException {
        ScannerSubscription subscription = new ScannerSubscription();
        // Test with a valid market cap
        double marketCap = 1000.50;
        subscription.marketCapBelow(marketCap);
        Field marketCapBelowField = ScannerSubscription.class.getDeclaredField("m_marketCapBelow");
        marketCapBelowField.setAccessible(true);
        // Using a tolerance for floating-point comparison
        assertEquals(marketCap, marketCapBelowField.getDouble(subscription), 0.001);
        // Test with a market cap of zero
        marketCap = 0;
        subscription.marketCapBelow(marketCap);
        assertEquals(marketCap, marketCapBelowField.getDouble(subscription), 0.001);
        // Test with a negative market cap
        marketCap = -100;
        subscription.marketCapBelow(marketCap);
        assertEquals(marketCap, marketCapBelowField.getDouble(subscription), 0.001);
        // Test with a market cap of Double.MAX_VALUE
        marketCap = Double.MAX_VALUE;
        subscription.marketCapBelow(marketCap);
        assertEquals(marketCap, marketCapBelowField.getDouble(subscription), 0.001);
    }
}
