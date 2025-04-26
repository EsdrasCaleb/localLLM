package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_couponRateBelow_17_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testCouponRateBelow_DefaultValue() {
        // Test the default value of m_couponRateBelow
        assertEquals(Double.MAX_VALUE, scannerSubscription.couponRateBelow(), "Default coupon rate below should be Double.MAX_VALUE");
    }

    @Test
    public void testCouponRateBelow_SetValue() {
        // Set a specific value for m_couponRateBelow and test it
        double expectedValue = 5.0;
        scannerSubscription.couponRateBelow(expectedValue);
        assertEquals(expectedValue, scannerSubscription.couponRateBelow(), "Coupon rate below should return the set value");
    }

    @Test
    public void testCouponRateBelow_NegativeValue() {
        // Set a negative value for m_couponRateBelow and test it
        double expectedValue = -1.0;
        scannerSubscription.couponRateBelow(expectedValue);
        assertEquals(expectedValue, scannerSubscription.couponRateBelow(), "Coupon rate below should return the set negative value");
    }
}
