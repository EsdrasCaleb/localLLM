package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapBelow_30_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testMarketCapBelow() throws NoSuchFieldException, IllegalAccessException {
        // Test with a valid market cap below value
        double testMarketCapBelow = 500000000.0;
        scannerSubscription.marketCapBelow(testMarketCapBelow);
        Field marketCapBelowField = ScannerSubscription.class.getDeclaredField("m_marketCapBelow");
        marketCapBelowField.setAccessible(true);
        double actualMarketCapBelow = (double) marketCapBelowField.get(scannerSubscription);
        assertEquals(testMarketCapBelow, actualMarketCapBelow, "The market cap below value should be set correctly");
        // Test with another valid market cap below value
        double anotherTestMarketCapBelow = 1000000000.0;
        scannerSubscription.marketCapBelow(anotherTestMarketCapBelow);
        actualMarketCapBelow = (double) marketCapBelowField.get(scannerSubscription);
        assertEquals(anotherTestMarketCapBelow, actualMarketCapBelow, "The market cap below value should be updated correctly");
        // Test with the maximum double value
        scannerSubscription.marketCapBelow(Double.MAX_VALUE);
        actualMarketCapBelow = (double) marketCapBelowField.get(scannerSubscription);
        assertEquals(Double.MAX_VALUE, actualMarketCapBelow, "The market cap below value should be set to Double.MAX_VALUE correctly");
        // Test with the minimum double value
        scannerSubscription.marketCapBelow(Double.MIN_VALUE);
        actualMarketCapBelow = (double) marketCapBelowField.get(scannerSubscription);
        assertEquals(Double.MIN_VALUE, actualMarketCapBelow, "The market cap below value should be set to Double.MIN_VALUE correctly");
        // Test with zero
        scannerSubscription.marketCapBelow(0.0);
        actualMarketCapBelow = (double) marketCapBelowField.get(scannerSubscription);
        assertEquals(0.0, actualMarketCapBelow, "The market cap below value should be set to 0.0 correctly");
        // Test with a negative value
        scannerSubscription.marketCapBelow(-1000000.0);
        actualMarketCapBelow = (double) marketCapBelowField.get(scannerSubscription);
        assertEquals(-1000000.0, actualMarketCapBelow, "The market cap below value should be set to a negative value correctly");
    }
}
