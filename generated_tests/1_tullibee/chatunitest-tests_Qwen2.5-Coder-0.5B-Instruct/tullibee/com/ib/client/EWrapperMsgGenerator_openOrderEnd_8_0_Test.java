package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

class EWrapperMsgGenerator_openOrderEnd_8_0_Test {

    @Test
    void testOpenOrderEnd() {
        // Arrange
        String expectedMessage = "END OF ORDER";
        // Act
        String actualMessage = EWrapperMsgGenerator.openOrderEnd();
        // Assert
        assertEquals(expectedMessage, actualMessage);
    }
}
