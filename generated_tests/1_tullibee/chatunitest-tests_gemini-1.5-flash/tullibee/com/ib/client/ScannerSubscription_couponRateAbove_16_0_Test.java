package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_couponRateAbove_16_0_Test {

    @Test
    void testCouponRateAbove_Default() {
        ScannerSubscription subscription = new ScannerSubscription();
        assertEquals(Double.MAX_VALUE, subscription.couponRateAbove());
    }

    @Test
    void testCouponRateAbove_SetAndGet() {
        ScannerSubscription subscription = new ScannerSubscription();
        double expectedCouponRate = 5.5;
        subscription.couponRateAbove(expectedCouponRate);
        assertEquals(expectedCouponRate, subscription.couponRateAbove());
    }

    @Test
    void testCouponRateAbove_Zero() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.couponRateAbove(0);
        assertEquals(0, subscription.couponRateAbove());
    }

    @Test
    void testCouponRateAbove_LargeValue() {
        ScannerSubscription subscription = new ScannerSubscription();
        double expectedCouponRate = 1000.0;
        subscription.couponRateAbove(expectedCouponRate);
        assertEquals(expectedCouponRate, subscription.couponRateAbove());
    }

    @Test
    void testCouponRateAbove_NegativeValue() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.couponRateAbove(-5.0);
        assertEquals(-5.0, subscription.couponRateAbove());
    }
}
