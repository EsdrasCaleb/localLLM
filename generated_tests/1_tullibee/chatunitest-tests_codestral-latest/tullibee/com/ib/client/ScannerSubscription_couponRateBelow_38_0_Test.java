package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_couponRateBelow_38_0_Test {

    @InjectMocks
    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testCouponRateBelow() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        double expectedCouponRateBelow = 5.5;
        // Act
        scannerSubscription.couponRateBelow(expectedCouponRateBelow);
        // Reflect to get the private field value
        Field field = ScannerSubscription.class.getDeclaredField("m_couponRateBelow");
        field.setAccessible(true);
        double actualCouponRateBelow = (double) field.get(scannerSubscription);
        // Assert
        assertEquals(expectedCouponRateBelow, actualCouponRateBelow, "The couponRateBelow should be set correctly");
    }
}
