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
    public void testScanCode() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        String testScanCode = "TestScanCode";
        // Act
        scannerSubscription.scanCode(testScanCode);
        // Reflect to get the private field value
        Field scanCodeField = ScannerSubscription.class.getDeclaredField("m_scanCode");
        scanCodeField.setAccessible(true);
        String actualScanCode = (String) scanCodeField.get(scannerSubscription);
        // Assert
        assertEquals(testScanCode, actualScanCode);
    }

    @Test
    public void testScanCodeWithNull() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        String testScanCode = null;
        // Act
        scannerSubscription.scanCode(testScanCode);
        // Reflect to get the private field value
        Field scanCodeField = ScannerSubscription.class.getDeclaredField("m_scanCode");
        scanCodeField.setAccessible(true);
        String actualScanCode = (String) scanCodeField.get(scannerSubscription);
        // Assert
        assertNull(actualScanCode);
    }
}
