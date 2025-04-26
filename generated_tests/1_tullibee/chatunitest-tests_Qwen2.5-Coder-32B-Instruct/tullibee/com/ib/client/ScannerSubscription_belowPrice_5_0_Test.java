package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_belowPrice_5_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testBelowPrice_DefaultValue() {
        // Default value of m_belowPrice is Double.MAX_VALUE
        assertEquals(Double.MAX_VALUE, scannerSubscription.belowPrice());
    }

    @Test
    public void testBelowPrice_SetValue() throws NoSuchFieldException, IllegalAccessException {
        // Set a new value to m_belowPrice using reflection
        Field belowPriceField = ScannerSubscription.class.getDeclaredField("m_belowPrice");
        belowPriceField.setAccessible(true);
        double testValue = 123.45;
        belowPriceField.set(scannerSubscription, testValue);
        // Verify that belowPrice() returns the new value
        assertEquals(testValue, scannerSubscription.belowPrice());
    }
}
