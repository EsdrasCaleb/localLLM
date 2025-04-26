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

public class EWrapperMsgGenerator_openOrderEnd_8_0_Test {

    @Test
    public void testOpenOrderEnd() throws Exception {
        // Create an instance of the class
        EWrapperMsgGenerator generator = new EWrapperMsgGenerator();
        // Use reflection to access the private method
        java.lang.reflect.Method method = EWrapperMsgGenerator.class.getDeclaredMethod("openOrderEnd");
        method.setAccessible(true);
        // Invoke the method and capture the result
        String result = (String) method.invoke(generator);
        // Assert the expected output
        assertEquals(" =============== end ===============", result);
    }
}
