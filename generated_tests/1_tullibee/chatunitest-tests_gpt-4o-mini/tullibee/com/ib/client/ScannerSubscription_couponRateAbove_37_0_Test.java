package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_couponRateAbove_37_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testCouponRateAbove() {
        // Test with a positive value
        scannerSubscription.couponRateAbove(5.0);
        assertEquals(5.0, invokePrivateField("m_couponRateAbove"), 0.001);
        // Test with zero
        scannerSubscription.couponRateAbove(0.0);
        assertEquals(0.0, invokePrivateField("m_couponRateAbove"), 0.001);
        // Test with a negative value
        scannerSubscription.couponRateAbove(-3.5);
        assertEquals(-3.5, invokePrivateField("m_couponRateAbove"), 0.001);
        // Test with maximum double value
        scannerSubscription.couponRateAbove(Double.MAX_VALUE);
        assertEquals(Double.MAX_VALUE, invokePrivateField("m_couponRateAbove"), 0.001);
    }

    // Helper method to invoke private fields using reflection
    private double invokePrivateField(String fieldName) {
        try {
            java.lang.reflect.Field field = ScannerSubscription.class.getDeclaredField(fieldName);
            field.setAccessible(true);
            return (double) field.get(scannerSubscription);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
