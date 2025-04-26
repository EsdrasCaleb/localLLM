// Test method
package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.text.DateFormat;

class EWrapperMsgGenerator_nextValidId_13_1_Test {

    @ExtendWith(MockitoExtension.class)
    public class TestClass {

        @Mock
        private EWrapperMsgGenerator wrapperMsgGenerator;

        @BeforeEach
        public void setUp() {
            MockitoAnnotations.initMocks(this);
        }

        @Test
        void nextValidIdTest() {
            // Create an instance of EWrapperMsgGenerator
            EWrapperMsgGenerator wrapperMsgGenerator = new EWrapperMsgGenerator();
            // Use Mockito to mock the nextValidId method
            when(wrapperMsgGenerator.nextValidId(123)).thenReturn("VALID_ID_123");
            // Call the method under test and verify the result
            String result = wrapperMsgGenerator.nextValidId(123);
            assert result.equals("VALID_ID_123");
        }
    }
}
