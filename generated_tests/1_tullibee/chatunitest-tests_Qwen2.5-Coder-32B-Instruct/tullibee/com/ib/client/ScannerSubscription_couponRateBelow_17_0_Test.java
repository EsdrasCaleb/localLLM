package com.ib.client;

import java.lang.reflect.Field;
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
    public void testCouponRateBelow_DefaultValue() throws NoSuchFieldException, IllegalAccessException {
        // Access the private field m_couponRateBelow using reflection
        Field couponRateBelowField = ScannerSubscription.class.getDeclaredField("m_couponRateBelow");
        couponRateBelowField.setAccessible(true);
        // Verify the default value
        double expectedDefaultValue = Double.MAX_VALUE;
        double actualDefaultValue = (double) couponRateBelowField.get(scannerSubscription);
        assertEquals(expectedDefaultValue, actualDefaultValue);
        // Verify the couponRateBelow() method returns the default value
        double result = scannerSubscription.couponRateBelow();
        assertEquals(expectedDefaultValue, result);
    }

    @Test
    public void testCouponRateBelow_SetValue() throws NoSuchFieldException, IllegalAccessException {
        // Set a new value using reflection
        Field couponRateBelowField = ScannerSubscription.class.getDeclaredField("m_couponRateBelow");
        couponRateBelowField.setAccessible(true);
        double newValue = 5.5;
        couponRateBelowField.set(scannerSubscription, newValue);
        // Verify the couponRateBelow() method returns the new value
        double result = scannerSubscription.couponRateBelow();
        assertEquals(newValue, result);
    }
}
