package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_couponRateBelow_38_0_Test {

    @Test
    void testCouponRateBelow() throws NoSuchFieldException, IllegalAccessException {
        ScannerSubscription subscription = new ScannerSubscription();
        // Test with a valid value
        double validRate = 5.5;
        subscription.couponRateBelow(validRate);
        assertEquals(validRate, subscription.couponRateBelow());
        // Test with Double.MAX_VALUE
        subscription.couponRateBelow(Double.MAX_VALUE);
        assertEquals(Double.MAX_VALUE, subscription.couponRateBelow());
        // Test with Double.MIN_VALUE
        subscription.couponRateBelow(Double.MIN_VALUE);
        assertEquals(Double.MIN_VALUE, subscription.couponRateBelow());
        // Test with zero
        subscription.couponRateBelow(0);
        assertEquals(0, subscription.couponRateBelow());
        // Test with a negative value
        subscription.couponRateBelow(-1.0);
        assertEquals(-1.0, subscription.couponRateBelow());
    }
}
