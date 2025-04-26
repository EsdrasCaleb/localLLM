package com.ib.client;

import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

public class EWrapperMsgGenerator_contractDetailsEnd_18_0_Test {

    private EWrapperMsgGenerator eWrapperMsgGenerator;

    @BeforeEach
    public void setUp() {
        eWrapperMsgGenerator = new EWrapperMsgGenerator();
    }

    @Test
    public void testContractDetailsEnd() throws Exception {
        // Arrange
        int reqId = 12345;
        String expectedMessage = "reqId = 12345 =============== end ===============";
        // Access the private method using reflection
        Method contractDetailsEndMethod = EWrapperMsgGenerator.class.getDeclaredMethod("contractDetailsEnd", int.class);
        contractDetailsEndMethod.setAccessible(true);
        // Act
        String result = (String) contractDetailsEndMethod.invoke(eWrapperMsgGenerator, reqId);
        // Assert
        assertEquals(expectedMessage, result);
    }
}
