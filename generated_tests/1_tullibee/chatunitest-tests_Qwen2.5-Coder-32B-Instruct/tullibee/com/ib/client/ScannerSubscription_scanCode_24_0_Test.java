package com.ib.client;

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
    public void testScanCodeSetsCorrectValue() {
        // Given
        String testScanCode = "TEST_CODE";
        // When
        scannerSubscription.scanCode(testScanCode);
        // Then
        assertEquals(testScanCode, getPrivateField(scannerSubscription, "m_scanCode"));
    }

    @Test
    public void testScanCodeSetsEmptyString() {
        // Given
        String testScanCode = "";
        // When
        scannerSubscription.scanCode(testScanCode);
        // Then
        assertEquals(testScanCode, getPrivateField(scannerSubscription, "m_scanCode"));
    }

    @Test
    public void testScanCodeSetsNullValue() {
        // Given
        String testScanCode = null;
        // When
        scannerSubscription.scanCode(testScanCode);
        // Then
        assertEquals(testScanCode, getPrivateField(scannerSubscription, "m_scanCode"));
    }

    private Object getPrivateField(final Object object, final String fieldName) {
        try {
            final var field = object.getClass().getDeclaredField(fieldName);
            field.setAccessible(true);
            return field.get(object);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }
}
