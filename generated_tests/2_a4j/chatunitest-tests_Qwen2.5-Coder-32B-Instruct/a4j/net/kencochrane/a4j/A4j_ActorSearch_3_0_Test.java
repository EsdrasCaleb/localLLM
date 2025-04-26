package net.kencochrane.a4j;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import net.kencochrane.a4j.DAO.Cart;
import net.kencochrane.a4j.DAO.Product;
import net.kencochrane.a4j.DAO.Search;
import net.kencochrane.a4j.beans.*;

@ExtendWith(MockitoExtension.class)
public class A4j_ActorSearch_3_0_Test {

    @Mock
    private Search search;

    @InjectMocks
    private A4j a4j;

    private static final String ACTOR_NAME = "Tom Hanks";

    private static final String MODE = "advanced";

    private static final String PAGE = "1";

    private ProductInfo expectedProductInfo;

    @BeforeEach
    public void setUp() {
        // Assuming ProductInfo has a default constructor
        expectedProductInfo = new ProductInfo();
        when(search.ActorSearch(ACTOR_NAME, MODE, PAGE)).thenReturn(expectedProductInfo);
    }

    @Test
    public void testActorSearch_Success() {
        ProductInfo result = a4j.ActorSearch(ACTOR_NAME, MODE, PAGE);
        assertNotNull(result);
        assertEquals(expectedProductInfo, result);
        verify(search, times(1)).ActorSearch(ACTOR_NAME, MODE, PAGE);
    }

    @Test
    public void testActorSearch_ModeNull() {
        when(search.ActorSearch(ACTOR_NAME, null, PAGE)).thenReturn(expectedProductInfo);
        ProductInfo result = a4j.ActorSearch(ACTOR_NAME, null, PAGE);
        assertNotNull(result);
        assertEquals(expectedProductInfo, result);
        verify(search, times(1)).ActorSearch(ACTOR_NAME, null, PAGE);
    }

    @Test
    public void testActorSearch_PageNull() {
        when(search.ActorSearch(ACTOR_NAME, MODE, null)).thenReturn(expectedProductInfo);
        ProductInfo result = a4j.ActorSearch(ACTOR_NAME, MODE, null);
        assertNotNull(result);
        assertEquals(expectedProductInfo, result);
        verify(search, times(1)).ActorSearch(ACTOR_NAME, MODE, null);
    }

    @Test
    public void testActorSearch_ActorNameNull() {
        when(search.ActorSearch(null, MODE, PAGE)).thenReturn(expectedProductInfo);
        ProductInfo result = a4j.ActorSearch(null, MODE, PAGE);
        assertNotNull(result);
        assertEquals(expectedProductInfo, result);
        verify(search, times(1)).ActorSearch(null, MODE, PAGE);
    }

    @Test
    public void testActorSearch_AllNull() {
        when(search.ActorSearch(null, null, null)).thenReturn(expectedProductInfo);
        ProductInfo result = a4j.ActorSearch(null, null, null);
        assertNotNull(result);
        assertEquals(expectedProductInfo, result);
        verify(search, times(1)).ActorSearch(null, null, null);
    }
}
