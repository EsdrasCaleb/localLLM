package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_couponRateAbove_16_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testCouponRateAbove_DefaultValue() {
        // Test the default value of m_couponRateAbove
        assertEquals(Double.MAX_VALUE, scannerSubscription.couponRateAbove(), "Default coupon rate above should be Double.MAX_VALUE");
    }

    @Test
    public void testCouponRateAbove_SetValue() {
        // Set a specific value and test if it returns the same
        double expectedValue = 5.0;
        scannerSubscription.couponRateAbove(expectedValue);
        assertEquals(expectedValue, scannerSubscription.couponRateAbove(), "couponRateAbove should return the set value");
    }

    @Test
    public void testCouponRateAbove_SetNegativeValue() {
        // Set a negative value and test if it returns the same
        double expectedValue = -1.0;
        scannerSubscription.couponRateAbove(expectedValue);
        assertEquals(expectedValue, scannerSubscription.couponRateAbove(), "couponRateAbove should return the set negative value");
    }

    @Test
    public void testCouponRateAbove_SetZeroValue() {
        // Set zero value and test if it returns the same
        double expectedValue = 0.0;
        scannerSubscription.couponRateAbove(expectedValue);
        assertEquals(expectedValue, scannerSubscription.couponRateAbove(), "couponRateAbove should return the set zero value");
    }
}
