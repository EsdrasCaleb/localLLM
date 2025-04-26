package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.HashMap;
import java.util.Map;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class AnyWrapperMsgGenerator_connectionClosed_3_2_Test {

    @Mock
    private AnyWrapperMsgGenerator anyWrapperMsgGenerator;

    @InjectMocks
    private AnyWrapperMsgGenerator anyWrapperMsgGeneratorMock;

    @Test
    public void test_connectionClosed() {
        // Arrange
        Map<String, String> map = new HashMap<>();
        map.put("id", "1");
        map.put("errorCode", "1");
        map.put("errorMsg", "Error message");
        // Act
        String result = anyWrapperMsgGenerator.connectionClosed();
        // Assert
        assertEquals("Connection Closed", result);
    }
}
