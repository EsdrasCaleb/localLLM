// Test class
package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

class Starring_getActor_3_3_Test {

    @ExtendWith(MockitoExtension.class)
    public class TestStarring {

        @Mock
        private Starring starring;

        @Test
        public void testGetActor() {
            // Arrange
            when(starring.getActor(1)).thenReturn("Bill Murray");
            // Act
            String result = starring.getActor(1);
            // Assert
            assertEquals("Bill Murray", result);
        }
    }
}
