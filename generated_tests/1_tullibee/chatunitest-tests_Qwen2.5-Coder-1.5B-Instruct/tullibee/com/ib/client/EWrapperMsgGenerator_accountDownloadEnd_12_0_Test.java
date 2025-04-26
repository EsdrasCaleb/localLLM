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

class EWrapperMsgGenerator_accountDownloadEnd_12_0_Test {

    @Mock
    private EWrapperMsgGenerator eWrapperMsgGenerator;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testAccountDownloadEnd() {
        // Arrange
        String expectedMessage = "accountDownloadEnd: exampleAccount";
        String actualMessage = eWrapperMsgGenerator.accountDownloadEnd("exampleAccount");
        // Act
        // Assert
        assertEquals(expectedMessage, actualMessage);
    }
}
