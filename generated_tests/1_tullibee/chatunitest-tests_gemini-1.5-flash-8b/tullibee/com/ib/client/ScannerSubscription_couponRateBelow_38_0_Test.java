package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_couponRateBelow_38_0_Test {

    @Test
    void testCouponRateBelow() throws NoSuchFieldException, IllegalAccessException {
        ScannerSubscription subscription = new ScannerSubscription();
        // Test case 1: Setting a valid coupon rate
        double couponRate = 5.5;
        subscription.couponRateBelow(couponRate);
        Field couponRateBelowField = ScannerSubscription.class.getDeclaredField("m_couponRateBelow");
        couponRateBelowField.setAccessible(true);
        assertEquals(couponRate, couponRateBelowField.getDouble(subscription), "Coupon rate not set correctly.");
        // Test case 2: Setting a coupon rate of zero
        couponRate = 0.0;
        subscription.couponRateBelow(couponRate);
        assertEquals(couponRate, couponRateBelowField.getDouble(subscription), "Coupon rate not set correctly.");
        // Test case 3: Setting a negative coupon rate
        couponRate = -2.5;
        subscription.couponRateBelow(couponRate);
        assertEquals(couponRate, couponRateBelowField.getDouble(subscription), "Coupon rate not set correctly.");
        // Test case 4: Setting a coupon rate of MAX_VALUE
        couponRate = Double.MAX_VALUE;
        subscription.couponRateBelow(couponRate);
        assertEquals(couponRate, couponRateBelowField.getDouble(subscription), "Coupon rate not set correctly.");
        // Test case 5: Setting a coupon rate of MIN_VALUE
        couponRate = Double.MIN_VALUE;
        subscription.couponRateBelow(couponRate);
        assertEquals(couponRate, couponRateBelowField.getDouble(subscription), "Coupon rate not set correctly.");
    }
}
