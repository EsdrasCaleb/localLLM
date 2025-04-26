package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_scanCode_3_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testScanCode_DefaultValue() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        Field scanCodeField = ScannerSubscription.class.getDeclaredField("m_scanCode");
        scanCodeField.setAccessible(true);
        // Set default value
        scanCodeField.set(scannerSubscription, null);
        // Act
        String result = scannerSubscription.scanCode();
        // Assert
        assertNull(result, "Expected scanCode to return null by default");
    }

    @Test
    public void testScanCode_SetValue() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        String expectedScanCode = "TEST_SCAN_CODE";
        Field scanCodeField = ScannerSubscription.class.getDeclaredField("m_scanCode");
        scanCodeField.setAccessible(true);
        scanCodeField.set(scannerSubscription, expectedScanCode);
        // Act
        String result = scannerSubscription.scanCode();
        // Assert
        assertEquals(expectedScanCode, result, "Expected scanCode to return the set value");
    }

    @Test
    public void testScanCode_UsingSetter() {
        // Arrange
        String expectedScanCode = "TEST_SCAN_CODE";
        scannerSubscription.scanCode(expectedScanCode);
        // Act
        String result = scannerSubscription.scanCode();
        // Assert
        assertEquals(expectedScanCode, result, "Expected scanCode to return the value set by the setter");
    }
}
