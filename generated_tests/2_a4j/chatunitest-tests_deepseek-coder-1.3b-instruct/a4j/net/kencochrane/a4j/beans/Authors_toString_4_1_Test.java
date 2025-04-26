package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Authors_toString_4_1_Test {

    private Authors authors;

    private ArrayList<String> mockAuthor;

    @BeforeEach
    public void setUp() {
        mockAuthor = Mockito.mock(ArrayList.class);
        authors = new Authors();
        authors.author = mockAuthor;
    }

    @Test
    public void testToString() {
        // given
        String expected = "# of Authors = 0\nAuthors is null or size 0\n";
        Mockito.when(mockAuthor.size()).thenReturn(0);
        Mockito.when(mockAuthor.get(0)).thenReturn(null);
        // when
        String result = authors.toString();
        // then
        assertEquals(expected, result);
    }

    @AfterEach
    public void tearDown() {
        authors = null;
    }
}
