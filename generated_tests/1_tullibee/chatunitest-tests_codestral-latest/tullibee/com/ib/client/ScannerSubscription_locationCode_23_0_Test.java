package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_locationCode_23_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testLocationCode() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        String expectedLocationCode = "NYSE";
        // Act
        scannerSubscription.locationCode(expectedLocationCode);
        // Assert
        Field field = ScannerSubscription.class.getDeclaredField("m_locationCode");
        field.setAccessible(true);
        String actualLocationCode = (String) field.get(scannerSubscription);
        assertEquals(expectedLocationCode, actualLocationCode);
    }

    @Test
    public void testLocationCode_Null() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        String expectedLocationCode = null;
        // Act
        scannerSubscription.locationCode(expectedLocationCode);
        // Assert
        Field field = ScannerSubscription.class.getDeclaredField("m_locationCode");
        field.setAccessible(true);
        String actualLocationCode = (String) field.get(scannerSubscription);
        assertNull(actualLocationCode);
    }
}
