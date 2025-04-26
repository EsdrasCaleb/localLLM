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

public class EWrapperMsgGenerator_nextValidId_13_0_Test {

    @Test
    public void testNextValidId() {
        // Arrange
        EWrapperMsgGenerator generator = new EWrapperMsgGenerator();
        // Act
        String result = generator.nextValidId(1);
        // Assert
        assertEquals("FA:1", result);
    }
}
