package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_scanCode_24_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testScanCodeSetsScanCode() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        String expectedScanCode = "TEST_SCAN_CODE";
        // Act
        scannerSubscription.scanCode(expectedScanCode);
        // Assert
        Field field = ScannerSubscription.class.getDeclaredField("m_scanCode");
        field.setAccessible(true);
        String actualScanCode = (String) field.get(scannerSubscription);
        assertEquals(expectedScanCode, actualScanCode);
    }

    @Test
    public void testScanCodeSetsNullScanCode() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        String expectedScanCode = null;
        // Act
        scannerSubscription.scanCode(expectedScanCode);
        // Assert
        Field field = ScannerSubscription.class.getDeclaredField("m_scanCode");
        field.setAccessible(true);
        String actualScanCode = (String) field.get(scannerSubscription);
        assertEquals(expectedScanCode, actualScanCode);
    }
}
