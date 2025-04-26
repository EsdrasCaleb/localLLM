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

public class EWrapperMsgGenerator_openOrderEnd_8_2_Test {

    @Test
    void testOpenOrderEnd() {
        // Arrange
        String expectedOutput = "END OF ORDER";
        // Act
        String actualOutput = EWrapperMsgGenerator.openOrderEnd();
        // Assert
        assertEquals(expectedOutput, actualOutput);
    }
}
