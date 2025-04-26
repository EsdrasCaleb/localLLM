package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_couponRateAbove_16_0_Test {

    @Test
    void testCouponRateAbove_positiveValue() {
        ScannerSubscription subscription = new ScannerSubscription();
        double expectedValue = 5.5;
        try {
            Field field = ScannerSubscription.class.getDeclaredField("m_couponRateAbove");
            field.setAccessible(true);
            field.set(subscription, expectedValue);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to access or set field: " + e.getMessage());
        }
        double actualValue = subscription.couponRateAbove();
        assertEquals(expectedValue, actualValue);
    }

    @Test
    void testCouponRateAbove_defaultValue() {
        ScannerSubscription subscription = new ScannerSubscription();
        double expectedValue = Double.MAX_VALUE;
        double actualValue = subscription.couponRateAbove();
        assertEquals(expectedValue, actualValue);
    }

    @Test
    void testCouponRateAbove_zeroValue() {
        ScannerSubscription subscription = new ScannerSubscription();
        double expectedValue = 0.0;
        try {
            Field field = ScannerSubscription.class.getDeclaredField("m_couponRateAbove");
            field.setAccessible(true);
            field.set(subscription, expectedValue);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to access or set field: " + e.getMessage());
        }
        double actualValue = subscription.couponRateAbove();
        assertEquals(expectedValue, actualValue);
    }
}
