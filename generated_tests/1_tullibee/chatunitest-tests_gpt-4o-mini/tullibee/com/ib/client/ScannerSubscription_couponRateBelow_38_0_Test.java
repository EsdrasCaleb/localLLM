package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_couponRateBelow_38_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testCouponRateBelow() {
        // Test with a normal value
        double testValue = 5.0;
        scannerSubscription.couponRateBelow(testValue);
        assertEquals(testValue, invokePrivateField("m_couponRateBelow"), "Coupon rate below should be set correctly.");
        // Test with zero
        testValue = 0.0;
        scannerSubscription.couponRateBelow(testValue);
        assertEquals(testValue, invokePrivateField("m_couponRateBelow"), "Coupon rate below should be set to zero.");
        // Test with negative value
        testValue = -3.0;
        scannerSubscription.couponRateBelow(testValue);
        assertEquals(testValue, invokePrivateField("m_couponRateBelow"), "Coupon rate below should be set to negative value.");
        // Test with maximum double value
        testValue = Double.MAX_VALUE;
        scannerSubscription.couponRateBelow(testValue);
        assertEquals(testValue, invokePrivateField("m_couponRateBelow"), "Coupon rate below should be set to maximum double value.");
    }

    private Object invokePrivateField(String fieldName) {
        try {
            java.lang.reflect.Field field = ScannerSubscription.class.getDeclaredField(fieldName);
            field.setAccessible(true);
            return field.get(scannerSubscription);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
