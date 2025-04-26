package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class EWrapperMsgGenerator_nextValidId_13_0_Test {

    private EWrapperMsgGenerator eWrapperMsgGenerator;

    @BeforeEach
    public void setUp() {
        eWrapperMsgGenerator = new EWrapperMsgGenerator();
    }

    @Test
    public void testNextValidId() throws Exception {
        // Given
        int orderId = 12345;
        String expectedMessage = "Next Valid Id: 12345";
        // When
        String result = invokePrivateMethod(eWrapperMsgGenerator, "nextValidId", orderId);
        // Then
        assertEquals(expectedMessage, result);
    }

    private String invokePrivateMethod(Object target, String methodName, Object... args) throws Exception {
        Class<?>[] argClasses = new Class<?>[args.length];
        for (int i = 0; i < args.length; i++) {
            if (args[i] instanceof Integer) {
                // Fixed: Use int.class instead of Integer.class for primitive int
                argClasses[i] = int.class;
            } else {
                argClasses[i] = args[i].getClass();
            }
        }
        java.lang.reflect.Method method = target.getClass().getDeclaredMethod(methodName, argClasses);
        method.setAccessible(true);
        return (String) method.invoke(target, args);
    }
}
