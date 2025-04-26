package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_maturityDateBelow_36_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testMaturityDateBelow() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        String testDate = "2023-12-31";
        // Act
        scannerSubscription.maturityDateBelow(testDate);
        // Assert
        Field field = ScannerSubscription.class.getDeclaredField("m_maturityDateBelow");
        field.setAccessible(true);
        String result = (String) field.get(scannerSubscription);
        assertEquals(testDate, result);
    }

    @Test
    public void testMaturityDateBelow_Null() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        String testDate = null;
        // Act
        scannerSubscription.maturityDateBelow(testDate);
        // Assert
        Field field = ScannerSubscription.class.getDeclaredField("m_maturityDateBelow");
        field.setAccessible(true);
        String result = (String) field.get(scannerSubscription);
        assertEquals(testDate, result);
    }

    @Test
    public void testMaturityDateBelow_EmptyString() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        String testDate = "";
        // Act
        scannerSubscription.maturityDateBelow(testDate);
        // Assert
        Field field = ScannerSubscription.class.getDeclaredField("m_maturityDateBelow");
        field.setAccessible(true);
        String result = (String) field.get(scannerSubscription);
        assertEquals(testDate, result);
    }
}
