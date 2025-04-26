package net.kencochrane.a4j.DAO;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import com.wutka.jox.JOXBeanInputStream;
import net.kencochrane.a4j.beans.BlendedSearch;
import net.kencochrane.a4j.beans.ProductInfo;
import net.kencochrane.a4j.beans.SellerSearch;
import net.kencochrane.a4j.file.FileUtil;
import java.io.FileInputStream;

@ExtendWith(MockitoExtension.class)
public class Search_Blended_0_3_Test {

    @Test
    public void testBlended() {
        BlendedSearch testBean = new BlendedSearch();
        FileUtil fileUtil = new FileUtil();
        FileInputStream fileIn = fileUtil.fetchBlendedSearchFile("apple", "json");
        JOXBeanInputStream joxIn = new JOXBeanInputStream(fileIn);
        try {
            testBean = (BlendedSearch) joxIn.readObject(BlendedSearch.class);
        } catch (Exception exc) {
            exc.printStackTrace();
        }
        assertNotNull(testBean);
    }
}
