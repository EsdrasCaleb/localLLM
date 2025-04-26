package net.kencochrane.a4j.DAO;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import com.wutka.jox.JOXBeanInputStream;
import net.kencochrane.a4j.beans.*;
import net.kencochrane.a4j.file.FileUtil;
import java.io.FileInputStream;
import java.util.ArrayList;

class Product_getProduct_0_0_Test {

    @InjectMocks
    private Product product;

    @Mock
    private FileUtil fileUtil;

    @Test
    public void getProductTest() throws Exception {
        MockitoAnnotations.openMocks(this);
        when(fileUtil.fetchASINFile(anyString(), anyString(), anyString(), anyString())).thenReturn(null);
        assertEquals(null, product.getProduct("1234567890", "offer", "page"));
    }
}
