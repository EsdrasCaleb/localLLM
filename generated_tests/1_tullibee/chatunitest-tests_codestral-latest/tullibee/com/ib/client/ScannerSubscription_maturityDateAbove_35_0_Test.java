package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_maturityDateAbove_35_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testMaturityDateAbove() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        String maturityDate = "2023-12-31";
        // Act
        scannerSubscription.maturityDateAbove(maturityDate);
        // Assert
        Field field = ScannerSubscription.class.getDeclaredField("m_maturityDateAbove");
        field.setAccessible(true);
        String actualMaturityDate = (String) field.get(scannerSubscription);
        assertEquals(maturityDate, actualMaturityDate);
    }

    @Test
    public void testMaturityDateAboveWithNull() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        String maturityDate = null;
        // Act
        scannerSubscription.maturityDateAbove(maturityDate);
        // Assert
        Field field = ScannerSubscription.class.getDeclaredField("m_maturityDateAbove");
        field.setAccessible(true);
        String actualMaturityDate = (String) field.get(scannerSubscription);
        assertNull(actualMaturityDate);
    }

    @Test
    public void testMaturityDateAboveWithEmptyString() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        String maturityDate = "";
        // Act
        scannerSubscription.maturityDateAbove(maturityDate);
        // Assert
        Field field = ScannerSubscription.class.getDeclaredField("m_maturityDateAbove");
        field.setAccessible(true);
        String actualMaturityDate = (String) field.get(scannerSubscription);
        assertEquals(maturityDate, actualMaturityDate);
    }
}
