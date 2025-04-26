package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_couponRateAbove_37_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    void testCouponRateAbove() throws NoSuchFieldException, IllegalAccessException {
        // Test with a positive value
        double testValue1 = 5.5;
        scannerSubscription.couponRateAbove(testValue1);
        assertEquals(testValue1, getPrivateField("m_couponRateAbove"), "Coupon rate above should be set to 5.5");
        // Test with zero
        double testValue2 = 0.0;
        scannerSubscription.couponRateAbove(testValue2);
        assertEquals(testValue2, getPrivateField("m_couponRateAbove"), "Coupon rate above should be set to 0.0");
        // Test with a negative value
        double testValue3 = -3.2;
        scannerSubscription.couponRateAbove(testValue3);
        assertEquals(testValue3, getPrivateField("m_couponRateAbove"), "Coupon rate above should be set to -3.2");
        // Test with MAX_VALUE
        double testValue4 = Double.MAX_VALUE;
        scannerSubscription.couponRateAbove(testValue4);
        assertEquals(testValue4, getPrivateField("m_couponRateAbove"), "Coupon rate above should be set to Double.MAX_VALUE");
        // Test with MIN_VALUE
        double testValue5 = Double.MIN_VALUE;
        scannerSubscription.couponRateAbove(testValue5);
        assertEquals(testValue5, getPrivateField("m_couponRateAbove"), "Coupon rate above should be set to Double.MIN_VALUE");
    }

    private double getPrivateField(String fieldName) throws NoSuchFieldException, IllegalAccessException {
        Field field = ScannerSubscription.class.getDeclaredField(fieldName);
        field.setAccessible(true);
        return (double) field.get(scannerSubscription);
    }
}
