package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_couponRateBelow_17_0_Test {

    @Test
    void testCouponRateBelow() {
        ScannerSubscription subscription = new ScannerSubscription();
        // Test default value
        assertEquals(Double.MAX_VALUE, subscription.couponRateBelow());
        // Test setting and retrieving a value
        double testValue = 10.5;
        subscription.couponRateBelow(testValue);
        assertEquals(testValue, subscription.couponRateBelow());
        // Test setting to a different value
        double testValue2 = 5.2;
        subscription.couponRateBelow(testValue2);
        assertEquals(testValue2, subscription.couponRateBelow());
        // Test setting to MAX_VALUE
        subscription.couponRateBelow(Double.MAX_VALUE);
        assertEquals(Double.MAX_VALUE, subscription.couponRateBelow());
        // Test setting to MIN_VALUE
        subscription.couponRateBelow(Double.MIN_VALUE);
        assertEquals(Double.MIN_VALUE, subscription.couponRateBelow());
        // Test setting to zero
        subscription.couponRateBelow(0);
        assertEquals(0, subscription.couponRateBelow());
        // Test setting to a negative value
        subscription.couponRateBelow(-5.0);
        assertEquals(-5.0, subscription.couponRateBelow());
    }
}
