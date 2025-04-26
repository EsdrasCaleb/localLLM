package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_couponRateBelow_17_0_Test {

    @Test
    void testCouponRateBelow_validInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        double expectedCouponRate = 5.5;
        try {
            Field couponRateBelowField = ScannerSubscription.class.getDeclaredField("m_couponRateBelow");
            couponRateBelowField.setAccessible(true);
            couponRateBelowField.set(subscription, expectedCouponRate);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to access or set the field: " + e.getMessage());
        }
        double actualCouponRate = subscription.couponRateBelow();
        assertEquals(expectedCouponRate, actualCouponRate);
    }

    @Test
    void testCouponRateBelow_defaultValue() {
        ScannerSubscription subscription = new ScannerSubscription();
        double expectedCouponRate = Double.MAX_VALUE;
        double actualCouponRate = subscription.couponRateBelow();
        assertEquals(expectedCouponRate, actualCouponRate);
    }
}
