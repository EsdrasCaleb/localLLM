package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_scannerSettingPairs_40_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testScannerSettingPairs() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        String expectedValue = "testValue";
        // Act
        scannerSubscription.scannerSettingPairs(expectedValue);
        // Reflect to get the private field value
        Field field = ScannerSubscription.class.getDeclaredField("m_scannerSettingPairs");
        field.setAccessible(true);
        String actualValue = (String) field.get(scannerSubscription);
        // Assert
        assertEquals(expectedValue, actualValue);
    }
}
