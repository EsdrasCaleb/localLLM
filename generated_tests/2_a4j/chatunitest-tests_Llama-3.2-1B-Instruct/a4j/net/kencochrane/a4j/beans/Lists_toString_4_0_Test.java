package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class Lists_toString_4_0_Test {

    @Mock
    private List<String> lists;

    @InjectMocks
    private Lists focal;

    @Test
    public void testToString() {
        // Arrange
        List<String> list = new ArrayList<>();
        list.add("Element 1");
        list.add("Element 2");
        list.add("Element 3");
        // Act
        String output = focal.toString();
        // Assert
        assertEquals("## of Lists = 3\nElement 1\nElement 2\nElement 3\n", output);
    }
}
