package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class Authors_toString_4_1_Test {

    @Mock
    private ArrayList<String> authors;

    @Mock
    private Authors authorsInstance;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
        authors = mock(ArrayList.class);
        authorsInstance = mock(Authors.class);
        when(authorsInstance.getAuthorsArray()).thenReturn(authors);
    }

    @Test
    public void testToString() {
        when(authorsInstance.getAuthorsArray()).thenReturn(authors);
        when(authors.get(0)).thenReturn("Alice");
        when(authors.get(1)).thenReturn("Bob");
        when(authors.get(2)).thenReturn("Charlie");
        String expectedOutput = "# of Authors = 3\nAuthor - Alice\nAuthor - Bob\nAuthor - Charlie";
        assertEquals(expectedOutput, authorsInstance.toString());
    }
}
