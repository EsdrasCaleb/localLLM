package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_couponRateAbove_37_0_Test {

    @Test
    void testCouponRateAbove() throws NoSuchFieldException, IllegalAccessException {
        ScannerSubscription subscription = new ScannerSubscription();
        // Test with a valid coupon rate
        double couponRate = 5.5;
        subscription.couponRateAbove(couponRate);
        assertEquals(couponRate, subscription.couponRateAbove());
        // Test with a zero coupon rate
        couponRate = 0.0;
        subscription.couponRateAbove(couponRate);
        assertEquals(couponRate, subscription.couponRateAbove());
        // Test with a negative coupon rate.
        couponRate = -2.5;
        subscription.couponRateAbove(couponRate);
        assertEquals(couponRate, subscription.couponRateAbove());
        // Test with a coupon rate close to Double.MAX_VALUE
        couponRate = Double.MAX_VALUE - 1000;
        subscription.couponRateAbove(couponRate);
        assertEquals(couponRate, subscription.couponRateAbove());
    }
}
