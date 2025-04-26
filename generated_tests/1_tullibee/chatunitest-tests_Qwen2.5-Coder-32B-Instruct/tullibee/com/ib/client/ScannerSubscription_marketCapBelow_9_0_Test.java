package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapBelow_9_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testMarketCapBelow_DefaultValue() {
        // Default value is Double.MAX_VALUE as per the class definition
        assertEquals(Double.MAX_VALUE, scannerSubscription.marketCapBelow());
    }

    @Test
    public void testMarketCapBelow_SetValue() throws Exception {
        double testValue = 1000000.0;
        Field marketCapBelowField = ScannerSubscription.class.getDeclaredField("m_marketCapBelow");
        marketCapBelowField.setAccessible(true);
        marketCapBelowField.set(scannerSubscription, testValue);
        assertEquals(testValue, scannerSubscription.marketCapBelow());
    }
}
