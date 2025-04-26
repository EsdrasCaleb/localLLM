package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_couponRateAbove_37_0_Test {

    @Test
    void testCouponRateAbove() {
        ScannerSubscription subscription = new ScannerSubscription();
        // Test with a valid positive value
        double validPositiveValue = 10.5;
        subscription.couponRateAbove(validPositiveValue);
        assertEquals(validPositiveValue, subscription.couponRateAbove());
        // Test with zero
        double zeroValue = 0.0;
        subscription.couponRateAbove(zeroValue);
        assertEquals(zeroValue, subscription.couponRateAbove());
        // Test with a valid negative value
        double validNegativeValue = -5.2;
        subscription.couponRateAbove(validNegativeValue);
        assertEquals(validNegativeValue, subscription.couponRateAbove());
        // Test with Double.MAX_VALUE
        subscription.couponRateAbove(Double.MAX_VALUE);
        assertEquals(Double.MAX_VALUE, subscription.couponRateAbove());
        // Test with Double.MIN_VALUE
        subscription.couponRateAbove(Double.MIN_VALUE);
        assertEquals(Double.MIN_VALUE, subscription.couponRateAbove());
        // Test with NaN
        subscription.couponRateAbove(Double.NaN);
        // NaN is not equal to itself
        assertNotEquals(Double.NaN, subscription.couponRateAbove());
        // Test with positive infinity
        subscription.couponRateAbove(Double.POSITIVE_INFINITY);
        assertEquals(Double.POSITIVE_INFINITY, subscription.couponRateAbove());
        // Test with negative infinity
        subscription.couponRateAbove(Double.NEGATIVE_INFINITY);
        assertEquals(Double.NEGATIVE_INFINITY, subscription.couponRateAbove());
    }
}
